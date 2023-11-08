'use client'
import type { FC } from 'react'
import React, { useState } from 'react'
import { useTranslation } from 'react-i18next'
import { useContext } from 'use-context-selector'
import Panel from '../base/feature-panel'
import ParamConfig from './param-config'
import ParamConfigContent from './param-config-content'
import { HelpCircle } from '@/app/components/base/icons/src/vender/line/general'
import Tooltip from '@/app/components/base/tooltip'
import Switch from '@/app/components/base/switch'
import { Eye } from '@/app/components/base/icons/src/vender/solid/general'
import ConfigContext from '@/context/debug-configuration'

const ConfigVision: FC = () => {
  const { t } = useTranslation()
  const [disabled, setDisabled] = useState(false)
  const {
    visionConfig,
  } = useContext(ConfigContext)

  const isShowVision = visionConfig.enable

  if (!isShowVision)
    return null

  return (<>
    <ParamConfigContent />
    <Panel
      className="mt-4"
      headerIcon={
        <Eye className='w-4 h-4 text-[#6938EF]'/>
      }
      title={
        <div className='flex items-center'>
          <div className='ml-1 mr-1'>{t('appDebug.vision.name')}</div>
          <Tooltip htmlContent={<div className='w-[180px]' >
            {t('appDebug.vision.description')}
          </div>} selector='config-vision-tooltip'>
            <HelpCircle className='w-[14px] h-[14px] text-gray-400' />
          </Tooltip>
        </div>
      }
      headerRight={
        <div className='flex items-center'>
          <ParamConfig />
          <div className='ml-4 mr-3 w-[1px] h-3.5 bg-gray-200'></div>
          <Switch
            defaultValue={disabled}
            onChange={setDisabled}
            size='md'
          />
        </div>
      }
      noBodySpacing
    />
  </>
  )
}
export default React.memo(ConfigVision)
